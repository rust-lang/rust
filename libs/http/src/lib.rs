#![no_std]
extern crate alloc;

use alloc::boxed::Box;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt::Write;
use core::str::FromStr;

use stem::syscall::vfs::{vfs_close, vfs_open, vfs_read, vfs_write};

pub struct TcpStream {
    data_fd: u32,
    ctl_fd: u32,
}

impl TcpStream {
    pub fn connect(host: &str, port: u16) -> Result<Self, String> {
        use abi::syscall::vfs_flags::{O_RDONLY, O_RDWR};

        // 1. Allocate a new TCP socket via /net/tcp/new
        let new_fd = vfs_open("/net/tcp/new", O_RDONLY).map_err(|e| format!("failed to open /net/tcp/new: {:?}", e))?;
        let mut buf = [0u8; 16];
        let n = vfs_read(new_fd, &mut buf).map_err(|e| format!("failed to read socket id: {:?}", e))?;
        let _ = vfs_close(new_fd);

        let socket_id = core::str::from_utf8(&buf[..n])
            .map_err(|_| "invalid socket id encoding")?
            .trim();

        // 2. Open ctl and data files
        let ctl_path = format!("/net/tcp/{}/ctl", socket_id);
        let data_path = format!("/net/tcp/{}/data", socket_id);

        let ctl_fd = vfs_open(&ctl_path, O_RDWR).map_err(|e| format!("failed to open ctl: {:?}", e))?;
        let data_fd = vfs_open(&data_path, O_RDWR).map_err(|e| format!("failed to open data: {:?}", e))?;

        // 3. Connect via ctl file
        let conn_cmd = format!("connect {} {}", host, port);
        vfs_write(ctl_fd, conn_cmd.as_bytes()).map_err(|e| format!("connect command failed: {:?}", e))?;

        // Wait for connection to establish (poor man's poll/check for now)
        // In a real implementation we would poll status or events.
        stem::time::sleep_ms(100);

        Ok(Self { data_fd, ctl_fd })
    }

    pub fn write(&mut self, data: &[u8]) -> Result<usize, String> {
        vfs_write(self.data_fd, data).map_err(|e| format!("write failed: {:?}", e))
    }

    pub fn read(&mut self, buf: &mut [u8]) -> Result<usize, String> {
        vfs_read(self.data_fd, buf).map_err(|e| {
            if e == abi::errors::Errno::EAGAIN {
                Ok(0)
            } else {
                Err(format!("read failed: {:?}", e))
            }
        })?
    }
}

impl Drop for TcpStream {
    fn drop(&mut self) {
        let _ = vfs_close(self.data_fd);
        let _ = vfs_close(self.ctl_fd);
    }
}

fn send_recv(netd_port: ChannelHandle, my_port: ChannelHandle, msg: &[u8]) -> Result<Vec<u8>, String> {
    if msg.len() < 2 {
        return Err("Invalid message".to_string());
    }
    let msg_type = u16::from_le_bytes([msg[0], msg[1]]);
    let payload = &msg[2..];

    // V2 Robust format: [4: response_port][8: caller_tid][2: msg_type][2: payload_len][payload...]
    let mut packet = Vec::with_capacity(16 + payload.len());
    packet.extend_from_slice(&(my_port as u32).to_le_bytes());
    let tid = stem::syscall::get_tid().unwrap_or(0);
    packet.extend_from_slice(&tid.to_le_bytes());
    packet.extend_from_slice(&msg_type.to_le_bytes());
    packet.extend_from_slice(&(payload.len() as u16).to_le_bytes());
    packet.extend_from_slice(payload);

    // Kernel IPC is capped at 4096 bytes.
    if packet.len() > 4096 {
        stem::warn!(
            "http: IPC message too large (len={}), will likely be truncated by kernel",
            packet.len()
        );
    }

    channel_send(netd_port, &packet).map_err(|_| "Send failed")?;

    let mut buf = [0u8; 8192]; // Large enough for response
    let start = stem::time::monotonic_ns();
    loop {
        match port_try_recv(my_port, &mut buf) {
            Ok(len) if len > 0 => return Ok(buf[..len].to_vec()),
            _ => {
                if stem::time::monotonic_ns() - start > 5_000_000_000 {
                    // 5s timeout
                    return Err("Timeout waiting for netd response".to_string());
                }
                stem::thread::yield_now();
            }
        }
    }
}

fn parse_ipv4(s: &str) -> Result<[u8; 4], ()> {
    let mut parts = s.split('.');
    let a = parts.next().ok_or(())?.parse::<u8>().map_err(|_| ())?;
    let b = parts.next().ok_or(())?.parse::<u8>().map_err(|_| ())?;
    let c = parts.next().ok_or(())?.parse::<u8>().map_err(|_| ())?;
    let d = parts.next().ok_or(())?.parse::<u8>().map_err(|_| ())?;
    if parts.next().is_some() {
        return Err(());
    }
    Ok([a, b, c, d])
}

// Minimal HTTP client
pub struct HttpClient;

impl HttpClient {
    pub fn post(url: &str, body: &str) -> Result<Response, String> {
        Self::request("POST", url, Some(body))
    }

    pub fn get(url: &str) -> Result<Response, String> {
        Self::request("GET", url, None)
    }

    fn request(method: &str, url: &str, body: Option<&str>) -> Result<Response, String> {
        let (host, port, path, final_url) = if url.starts_with("http://") {
            let rest = &url[7..];
            let (host_port, path) = if let Some(idx) = rest.find('/') {
                (&rest[..idx], &rest[idx..])
            } else {
                (rest, "/")
            };

            let (host, port) = if let Some(idx) = host_port.find(':') {
                (
                    &host_port[..idx],
                    host_port[idx + 1..]
                        .parse::<u16>()
                        .map_err(|_| "Invalid port")?,
                )
            } else {
                (host_port, 80)
            };
            (host.to_string(), port, path.to_string(), url.to_string())
        } else {
            // Use proxy for non-http (likely https)
            // Proxy format: http://10.0.2.2:8081/?url=<encoded_url>
            // Note: 10.0.2.2 is QEMU host loopback
            let encoded_url = url_encode(url);
            let proxy_path = format!("/?url={}", encoded_url);
            ("10.0.2.2".to_string(), 8081, proxy_path, url.to_string())
        };

        let mut stream = TcpStream::connect(&host, port)?;

        let mut req = String::new();
        write!(req, "{} {} HTTP/1.1\r\n", method, path).ok();
        write!(req, "Host: {}\r\n", host).ok();
        write!(req, "Connection: close\r\n").ok();
        if host == "10.0.2.2" {
            write!(req, "X-Original-URL: {}\r\n", final_url).ok();
        }
        if let Some(b) = body {
            write!(req, "Content-Length: {}\r\n", b.len()).ok();
            write!(req, "Content-Type: application/json\r\n").ok();
        }
        write!(req, "\r\n").ok();
        if let Some(b) = body {
            req.push_str(b);
        }

        stream.write(req.as_bytes())?;

        let mut buffer = Vec::new();
        let mut temp_buf = [0u8; 1024];
        let mut body_start = 0;
        let mut headers_done = false;

        // Initial read loop to find headers
        for _ in 0..20 {
            // Limit tries
            let n = stream.read(&mut temp_buf)?;
            if n == 0 {
                break;
            }
            buffer.extend_from_slice(&temp_buf[..n]);

            if let Some(idx) = find_subsequence(&buffer, b"\r\n\r\n") {
                body_start = idx + 4;
                headers_done = true;
                break;
            }
        }

        if !headers_done {
            // Maybe no body or something weird, but let's assume we have what we have
        }

        Ok(Response {
            stream,
            buffer,
            cursor: body_start,
        })
    }
}

pub struct Response {
    stream: TcpStream,
    buffer: Vec<u8>,
    cursor: usize,
}

impl Response {
    pub fn read_chunk(&mut self) -> Result<Vec<u8>, String> {
        if self.cursor < self.buffer.len() {
            let chunk = self.buffer[self.cursor..].to_vec();
            self.cursor = self.buffer.len();
            return Ok(chunk);
        }

        let mut buf = [0u8; 1024];
        let n = self.stream.read(&mut buf)?;
        if n == 0 {
            return Ok(Vec::new());
        }
        Ok(buf[..n].to_vec())
    }
}

fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn url_encode(s: &str) -> String {
    let mut out = String::new();
    for b in s.as_bytes() {
        if b.is_ascii_alphanumeric() || b"-_.~".contains(b) {
            out.push(*b as char);
        } else {
            write!(out, "%{:02X}", b).ok();
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate std;

    #[test]
    fn test_url_encoding() {
        // Alphanumeric - should not be encoded
        assert_eq!(url_encode("abc123XYZ"), "abc123XYZ");

        // Allowed characters - should not be encoded
        assert_eq!(url_encode("a-b_c.d~e"), "a-b_c.d~e");

        // Space - should be encoded as %20
        assert_eq!(url_encode("hello world"), "hello%20world");

        // Special characters - should be encoded
        // / -> %2F, : -> %3A
        assert_eq!(url_encode("http://example.com"), "http%3A%2F%2Fexample.com");

        // Empty string
        assert_eq!(url_encode(""), "");
    }

    #[test]
    fn test_parse_ipv4() {
        // Valid IPs
        assert_eq!(parse_ipv4("127.0.0.1"), Ok([127, 0, 0, 1]));
        assert_eq!(parse_ipv4("192.168.1.100"), Ok([192, 168, 1, 100]));
        assert_eq!(parse_ipv4("0.0.0.0"), Ok([0, 0, 0, 0]));
        assert_eq!(parse_ipv4("255.255.255.255"), Ok([255, 255, 255, 255]));

        // Invalid format
        assert_eq!(parse_ipv4(""), Err(()));
        assert_eq!(parse_ipv4("1.2.3"), Err(()));
        assert_eq!(parse_ipv4("1.2.3.4.5"), Err(()));

        // Invalid numbers
        assert_eq!(parse_ipv4("256.0.0.1"), Err(()));
        assert_eq!(parse_ipv4("-1.0.0.0"), Err(()));

        // Non-numeric
        assert_eq!(parse_ipv4("a.b.c.d"), Err(()));
    }

    #[test]
    fn test_find_subsequence() {
        let sep = b"\r\n\r\n";

        // Found at end (typical header case)
        let data = b"Hello world\r\n\r\n";
        assert_eq!(find_subsequence(data, sep), Some(11));

        // Found in middle
        let data2 = b"Hello\r\n\r\nBody";
        assert_eq!(find_subsequence(data2, sep), Some(5));

        // Not found
        let data3 = b"Hello world";
        assert_eq!(find_subsequence(data3, sep), None);

        // Found at start
        let data4 = b"\r\n\r\nStart";
        assert_eq!(find_subsequence(data4, sep), Some(0));

        // Partial match
        let data5 = b"Partial\r\n\rEnd";
        assert_eq!(find_subsequence(data5, sep), None);

        // Overlapping needle
        assert_eq!(find_subsequence(b"aaaaa", b"aa"), Some(0));
    }
}
