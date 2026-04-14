//! HTTP/1.1 client with redirect support
#![no_std]
extern crate alloc;
use alloc::string::ToString;
use core::default::Default;

use alloc::string::String;
use alloc::vec::Vec;
use smoltcp::iface::Interface;
use smoltcp::socket::tcp::{Socket as TcpSocket, SocketBuffer};
use smoltcp::time::Duration;
use smoltcp::wire::{IpAddress, IpEndpoint, Ipv4Address};

use crate::dns;
use crate::smol_device::VirtioNicDevice;

/// Maximum number of redirects to follow
const MAX_REDIRECTS: u8 = 5;

/// HTTPS proxy server (QEMU user-mode networking gateway)
/// The host runs scripts/https_proxy.py on port 8081
const PROXY_HOST: Ipv4Address = Ipv4Address::new(10, 0, 2, 2);
const PROXY_PORT: u16 = 8081;

#[derive(Debug)]
pub enum HttpError {
    Timeout,
    ConnectionFailed,
    InvalidResponse,
    TooManyRedirects,
    HttpsRequired,
    DnsLookupFailed,
}

pub struct HttpResponse {
    pub status_code: u16,
    pub body: Vec<u8>,
}

/// Top-level HTTP GET that follows redirects
pub fn http_get(
    iface: &mut Interface,
    device: &mut VirtioNicDevice<'_>,
    dns_server: Ipv4Address,
    ip: Ipv4Address,
    host: &str,
    path: &str,
) -> Result<HttpResponse, HttpError> {
    http_get_internal(iface, device, dns_server, ip, host, path, 0)
}

/// Internal HTTP GET with redirect counter
fn http_get_internal(
    iface: &mut Interface,
    device: &mut VirtioNicDevice<'_>,
    dns_server: Ipv4Address,
    ip: Ipv4Address,
    host: &str,
    path: &str,
    redirect_count: u8,
) -> Result<HttpResponse, HttpError> {
    if redirect_count > MAX_REDIRECTS {
        return Err(HttpError::TooManyRedirects);
    }

    let mut rx_data = [0u8; 8192];
    let mut tx_data = [0u8; 2048];
    
    let tcp_rx_buffer = SocketBuffer::new(&mut rx_data[..]);
    let tcp_tx_buffer = SocketBuffer::new(&mut tx_data[..]);
    let mut tcp_socket = TcpSocket::new(tcp_rx_buffer, tcp_tx_buffer);

    let local_port = 49152 + (stem::time::monotonic_ns() % 16384) as u16;
    tcp_socket.set_timeout(Some(Duration::from_secs(10)));

    let mut sockets_storage: [smoltcp::iface::SocketStorage; 1] = Default::default();
    let mut socket_set = smoltcp::iface::SocketSet::new(&mut sockets_storage[..]);
    let tcp_handle = socket_set.add(tcp_socket);
    
    let endpoint = IpEndpoint::new(IpAddress::Ipv4(ip), 80);

    stem::info!("HTTP: GET http://{}{}  (redirect #{})", host, path, redirect_count);

    let start = VirtioNicDevice::now();
    let timeout = start + Duration::from_secs(30);

    // Connect
    let socket = socket_set.get_mut::<TcpSocket>(tcp_handle);
    socket
        .connect(iface.context(), endpoint, local_port)
        .map_err(|_| HttpError::ConnectionFailed)?;

    let mut connected = false;
    let mut request_sent = false;
    let mut response_data = Vec::new();
    let mut headers_complete = false;
    let mut status_code = 0u16;
    #[allow(unused_assignments)]
    let mut content_length: Option<usize> = None;
    let mut chunked = false;
    let mut location_header: Option<String> = None;

    loop {
        let now = VirtioNicDevice::now();
        if now > timeout {
            return Err(HttpError::Timeout);
        }

        iface.poll(now, device, &mut socket_set);

        let socket = socket_set.get_mut::<TcpSocket>(tcp_handle);

        if !connected && socket.may_send() {
            connected = true;
        }

        if connected && !request_sent && socket.can_send() {
            let request = alloc::format!(
                "GET {} HTTP/1.1\r\nHost: {}\r\nConnection: close\r\n\r\n",
                path, host
            );
            socket.send_slice(request.as_bytes()).ok();
            request_sent = true;
        }

        if request_sent && socket.can_recv() {
            let data = socket.recv(|buffer| {
                let len = buffer.len();
                response_data.extend_from_slice(buffer);
                (len, ())
            }).ok();

            if data.is_some() && !headers_complete {
                if let Some(end_of_headers) = find_pattern(&response_data, b"\r\n\r\n") {
                    headers_complete = true;
                    let headers = &response_data[..end_of_headers];

                    // Parse status code
                    if let Some(status) = parse_status_code(headers) {
                        status_code = status;
                        stem::info!("HTTP: Status {}", status_code);

                        // Check for redirect and parse Location
                        if status_code >= 300 && status_code < 400 {
                            if let Some(loc_bytes) = parse_header(headers, b"Location") {
                                if let Ok(loc_str) = core::str::from_utf8(loc_bytes) {
                                    location_header = Some(String::from(loc_str));
                                }
                            }
                        }
                    }

                    // Parse Content-Length
                    content_length = parse_header(headers, b"Content-Length")
                        .and_then(|v| core::str::from_utf8(v).ok())
                        .and_then(|v| v.parse().ok());

                    // Check for chunked encoding
                    if let Some(encoding) = parse_header(headers, b"Transfer-Encoding") {
                        if encoding == b"chunked" {
                            chunked = true;
                        }
                    }
                }
            }
        }

        if !socket.is_open() {
            break;
        }

        stem::time::sleep_ms(10);
    }

    if !headers_complete {
        return Err(HttpError::InvalidResponse);
    }

    // Handle redirect
    if status_code >= 300 && status_code < 400 {
        if let Some(location) = location_header {
            return follow_redirect(iface, device, dns_server, host, &location, redirect_count);
        }
        // No Location header in redirect - invalid response
        return Err(HttpError::InvalidResponse);
    }

    // Extract body
    let body = if let Some(end_of_headers) = find_pattern(&response_data, b"\r\n\r\n") {
        let body_start = end_of_headers + 4;
        let mut body_data = response_data[body_start..].to_vec();

        // Handle chunked encoding
        if chunked {
            body_data = decode_chunked(&body_data);
        }

        body_data
    } else {
        Vec::new()
    };

    stem::info!("HTTP: Response complete ({} bytes)", body.len());

    Ok(HttpResponse { status_code, body })
}

/// Follow a redirect to a new location
fn follow_redirect(
    iface: &mut Interface,
    device: &mut VirtioNicDevice<'_>,
    dns_server: Ipv4Address,
    current_host: &str,
    location: &str,
    redirect_count: u8,
) -> Result<HttpResponse, HttpError> {
    stem::info!("HTTP: Following redirect to {}", location);

    // Check for HTTPS - route through proxy
    if location.starts_with("https://") {
        stem::info!("HTTP: HTTPS redirect detected, using proxy at {}:{}", PROXY_HOST, PROXY_PORT);
        return fetch_via_proxy(iface, device, dns_server, location, redirect_count);
    }

    // Parse the Location URL
    let (new_host, new_path) = if location.starts_with("http://") {
        // Absolute HTTP URL
        let without_scheme = &location[7..];
        parse_host_path(without_scheme)
    } else if location.starts_with("/") {
        // Relative path - keep same host
        (String::from(current_host), String::from(location))
    } else {
        // Relative path without leading slash
        (String::from(current_host), alloc::format!("/{}", location))
    };

    stem::info!("HTTP: Redirect target: host={} path={}", new_host, new_path);

    // If host changed, we need DNS lookup
    let new_ip = if new_host.as_str() != current_host {
        // Need to resolve the new host
        stem::info!("HTTP: Resolving new host {}", new_host);
        match dns::lookup_a(iface, device, dns_server, &new_host) {
            Ok(ip) => {
                stem::info!("HTTP: Resolved {} to {}", new_host, ip);
                ip
            }
            Err(_) => {
                stem::warn!("HTTP: DNS lookup failed for {}", new_host);
                return Err(HttpError::DnsLookupFailed);
            }
        }
    } else {
        // Same host - we'd need to keep track of the IP, but for simplicity
        // we'll just re-resolve it
        match dns::lookup_a(iface, device, dns_server, &new_host) {
            Ok(ip) => ip,
            Err(_) => return Err(HttpError::DnsLookupFailed),
        }
    };

    // Recurse with incremented redirect count
    http_get_internal(iface, device, dns_server, new_ip, &new_host, &new_path, redirect_count + 1)
}

/// Fetch a URL via the HTTPS proxy server
/// The proxy runs on the host at 10.0.2.2:8081 and accepts requests like:
///   GET /?url=https://example.com/path
fn fetch_via_proxy(
    iface: &mut Interface,
    device: &mut VirtioNicDevice<'_>,
    _dns_server: Ipv4Address,
    target_url: &str,
    redirect_count: u8,
) -> Result<HttpResponse, HttpError> {
    if redirect_count > MAX_REDIRECTS {
        return Err(HttpError::TooManyRedirects);
    }

    // URL-encode the target URL for the query parameter
    let mut encoded_url = String::new();
    for byte in target_url.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                encoded_url.push(byte as char);
            }
            _ => {
                encoded_url.push('%');
                encoded_url.push(char::from_digit((byte >> 4) as u32, 16).unwrap_or('0'));
                encoded_url.push(char::from_digit((byte & 0xF) as u32, 16).unwrap_or('0'));
            }
        }
    }

    let proxy_path = alloc::format!("/?url={}", encoded_url);
    stem::info!("HTTP: Proxy request: GET http://{}:{}{}", PROXY_HOST, PROXY_PORT, proxy_path);

    let mut rx_data = [0u8; 32768]; // Larger buffer for proxied content
    let mut tx_data = [0u8; 2048];
    
    let tcp_rx_buffer = SocketBuffer::new(&mut rx_data[..]);
    let tcp_tx_buffer = SocketBuffer::new(&mut tx_data[..]);
    let mut tcp_socket = TcpSocket::new(tcp_rx_buffer, tcp_tx_buffer);

    let local_port = 49152 + (stem::time::monotonic_ns() % 16384) as u16;
    tcp_socket.set_timeout(Some(Duration::from_secs(30)));

    let mut sockets_storage: [smoltcp::iface::SocketStorage; 1] = Default::default();
    let mut socket_set = smoltcp::iface::SocketSet::new(&mut sockets_storage[..]);
    let tcp_handle = socket_set.add(tcp_socket);
    
    let endpoint = IpEndpoint::new(IpAddress::Ipv4(PROXY_HOST), PROXY_PORT);

    let start = VirtioNicDevice::now();
    let timeout = start + Duration::from_secs(60); // Longer timeout for proxy

    // Connect to proxy
    let socket = socket_set.get_mut::<TcpSocket>(tcp_handle);
    socket
        .connect(iface.context(), endpoint, local_port)
        .map_err(|_| HttpError::ConnectionFailed)?;

    let mut connected = false;
    let mut request_sent = false;
    let mut response_data = Vec::new();
    let mut headers_complete = false;
    let mut status_code = 0u16;
    let mut content_length: Option<usize> = None;
    let mut chunked = false;

    loop {
        let now = VirtioNicDevice::now();
        if now > timeout {
            return Err(HttpError::Timeout);
        }

        iface.poll(now, device, &mut socket_set);

        let socket = socket_set.get_mut::<TcpSocket>(tcp_handle);

        if !connected && socket.may_send() {
            connected = true;
            stem::info!("HTTP: Connected to proxy");
        }

        if connected && !request_sent && socket.can_send() {
            let request = alloc::format!(
                "GET {} HTTP/1.1\r\nHost: {}:{}\r\nConnection: close\r\n\r\n",
                proxy_path, PROXY_HOST, PROXY_PORT
            );
            socket.send_slice(request.as_bytes()).ok();
            request_sent = true;
            stem::info!("HTTP: Proxy request sent");
        }

        if request_sent && socket.can_recv() {
            let _ = socket.recv(|buffer| {
                let len = buffer.len();
                response_data.extend_from_slice(buffer);
                (len, ())
            });

            if !headers_complete {
                if let Some(end_of_headers) = find_pattern(&response_data, b"\r\n\r\n") {
                    headers_complete = true;
                    let headers = &response_data[..end_of_headers];

                    if let Some(status) = parse_status_code(headers) {
                        status_code = status;
                        stem::info!("HTTP: Proxy response status {}", status_code);
                    }

                    content_length = parse_header(headers, b"Content-Length")
                        .and_then(|v| core::str::from_utf8(v).ok())
                        .and_then(|v| v.parse().ok());

                    if let Some(encoding) = parse_header(headers, b"Transfer-Encoding") {
                        if encoding == b"chunked" {
                            chunked = true;
                        }
                    }

                    stem::info!("HTTP: Proxy headers complete (cl={:?}, chunked={})", content_length, chunked);
                }
            }
        }

        if !socket.is_open() {
            break;
        }

        stem::time::sleep_ms(10);
    }

    if !headers_complete {
        stem::warn!("HTTP: Proxy response incomplete");
        return Err(HttpError::InvalidResponse);
    }

    // Extract body
    let body = if let Some(end_of_headers) = find_pattern(&response_data, b"\r\n\r\n") {
        let body_start = end_of_headers + 4;
        let mut body_data = response_data[body_start..].to_vec();

        if chunked {
            body_data = decode_chunked(&body_data);
        }

        body_data
    } else {
        Vec::new()
    };

    stem::info!("HTTP: Proxy fetch complete ({} bytes, status {})", body.len(), status_code);

    Ok(HttpResponse { status_code, body })
}

/// Parse host and path from a URL fragment like "example.com/foo/bar"
fn parse_host_path(url_fragment: &str) -> (String, String) {
    if let Some(slash_pos) = url_fragment.find('/') {
        let host = &url_fragment[..slash_pos];
        let path = &url_fragment[slash_pos..];
        (String::from(host), String::from(path))
    } else {
        // No path - default to /
        (String::from(url_fragment), String::from("/"))
    }
}

fn find_pattern(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn parse_status_code(headers: &[u8]) -> Option<u16> {
    let first_line = headers.split(|&b| b == b'\n').next()?;
    let parts: Vec<&[u8]> = first_line.splitn(3, |&b| b == b' ').collect();
    if parts.len() >= 2 {
        core::str::from_utf8(parts[1])
            .ok()
            .and_then(|s| s.parse().ok())
    } else {
        None
    }
}

fn parse_header<'a>(headers: &'a [u8], name: &[u8]) -> Option<&'a [u8]> {
    for line in headers.split(|&b| b == b'\n') {
        if line.len() > name.len() + 2 {
            let (key, rest) = line.split_at(name.len());
            if key.eq_ignore_ascii_case(name) && rest.starts_with(b":") {
                let value_start = rest.iter().position(|&b| b != b':' && b != b' ')?;
                let value = &rest[value_start..];
                let value_end = value.iter().position(|&b| b == b'\r').unwrap_or(value.len());
                return Some(&value[..value_end]);
            }
        }
    }
    None
}

fn decode_chunked(data: &[u8]) -> Vec<u8> {
    let mut result = Vec::new();
    let mut pos = 0;

    while pos < data.len() {
        // Find chunk size line
        let chunk_line_end = data[pos..]
            .windows(2)
            .position(|w| w == b"\r\n")
            .map(|p| pos + p);

        if let Some(end) = chunk_line_end {
            let size_str = core::str::from_utf8(&data[pos..end]).ok();
            if let Some(size_hex) = size_str {
                if let Ok(size) = usize::from_str_radix(size_hex.trim(), 16) {
                    if size == 0 {
                        break;
                    }
                    let chunk_start = end + 2;
                    let chunk_end = chunk_start + size;
                    if chunk_end <= data.len() {
                        result.extend_from_slice(&data[chunk_start..chunk_end]);
                        pos = chunk_end + 2; // Skip trailing \r\n
                        continue;
                    }
                }
            }
        }
        break;
    }

    result
}
