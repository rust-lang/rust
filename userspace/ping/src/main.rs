//! Simple ping utility.
//!
//! Since ICMP raw sockets are not yet available through the /net/ VFS,
//! this implementation probes connectivity by opening TCP connections to
//! the target host (default port 80) and measuring the round-trip time.
//! Usage: ping [-c count] [-p port] <host>
#![no_std]
#![no_main]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;


use alloc::string::String;
use alloc::vec::Vec;
use abi::syscall::vfs_flags::{O_RDONLY, O_WRONLY};
use stem::syscall::{argv_get, vfs_close, vfs_open, vfs_read, vfs_write};

fn get_args() -> Vec<String> {
    let mut len = 0;
    if let Ok(l) = argv_get(&mut []) {
        len = l;
    }
    if len == 0 {
        return Vec::new();
    }
    let mut buf = alloc::vec![0u8; len];
    if argv_get(&mut buf).is_err() {
        return Vec::new();
    }

    let mut args = Vec::new();
    if buf.len() >= 4 {
        let count = u32::from_le_bytes(buf[0..4].try_into().unwrap()) as usize;
        let mut offset = 4;
        for _ in 0..count {
            if offset + 4 > buf.len() {
                break;
            }
            let str_len =
                u32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap()) as usize;
            offset += 4;
            if offset + str_len > buf.len() {
                break;
            }
            if let Ok(s) = core::str::from_utf8(&buf[offset..offset + str_len]) {
                args.push(String::from(s));
            }
            offset += str_len;
        }
    }
    args
}

fn print(fd: u32, s: &str) {
    let _ = vfs_write(fd, s.as_bytes());
}

fn read_file(path: &str) -> String {
    let Ok(fd) = vfs_open(path, O_RDONLY) else {
        return String::new();
    };
    let mut buf = alloc::vec![0u8; 128];
    let n = vfs_read(fd, &mut buf).unwrap_or(0);
    let _ = vfs_close(fd);
    buf.truncate(n);
    String::from_utf8_lossy(&buf).trim().into()
}

/// Build a minimal DNS A-record query.
fn build_dns_query(name: &str) -> Vec<u8> {
    let mut pkt: Vec<u8> = Vec::new();
    pkt.extend_from_slice(&[0xAB, 0xCD]);
    pkt.extend_from_slice(&[0x01, 0x00]);
    pkt.extend_from_slice(&[0x00, 0x01]);
    pkt.extend_from_slice(&[0x00, 0x00, 0x00, 0x00, 0x00, 0x00]);
    for label in name.split('.') {
        pkt.push(label.len() as u8);
        pkt.extend_from_slice(label.as_bytes());
    }
    pkt.push(0);
    pkt.extend_from_slice(&[0x00, 0x01, 0x00, 0x01]);
    pkt
}

/// Parse the first A record from a DNS response.
fn parse_first_a(data: &[u8]) -> Option<[u8; 4]> {
    if data.len() < 12 {
        return None;
    }
    let ancount = u16::from_be_bytes([data[6], data[7]]) as usize;
    if ancount == 0 {
        return None;
    }
    let mut pos = 12;
    // Skip question QNAME
    loop {
        if pos >= data.len() {
            return None;
        }
        let len = data[pos] as usize;
        if len == 0 {
            pos += 1;
            break;
        }
        if len & 0xC0 == 0xC0 {
            pos += 2;
            break;
        }
        pos += 1 + len;
    }
    pos += 4; // QTYPE + QCLASS
    for _ in 0..ancount {
        if pos >= data.len() {
            break;
        }
        let b = data[pos];
        if b & 0xC0 == 0xC0 {
            pos += 2;
        } else {
            loop {
                if pos >= data.len() {
                    return None;
                }
                let l = data[pos] as usize;
                pos += 1;
                if l == 0 {
                    break;
                }
                pos += l;
            }
        }
        if pos + 10 > data.len() {
            break;
        }
        let rtype = u16::from_be_bytes([data[pos], data[pos + 1]]);
        let rdlen = u16::from_be_bytes([data[pos + 8], data[pos + 9]]) as usize;
        pos += 10;
        if pos + rdlen > data.len() {
            break;
        }
        if rtype == 1 && rdlen == 4 {
            return Some([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
        }
        pos += rdlen;
    }
    None
}

/// Resolve a hostname to an IPv4 address using /net/dns/server for the resolver.
/// If the name is already a dotted-decimal IP, return it unchanged.
fn resolve(name: &str) -> Result<String, &'static str> {
    // Check if already an IP address
    if name.split('.').count() == 4 && name.chars().all(|c| c.is_ascii_digit() || c == '.') {
        return Ok(String::from(name));
    }

    let dns_server = {
        let s = read_file("/net/dns/server");
        if s.is_empty() || s == "0.0.0.0" {
            String::from("8.8.8.8")
        } else {
            s
        }
    };

    // Allocate UDP socket
    let Ok(nfd) = vfs_open("/net/udp/new", O_RDONLY) else {
        return Err("cannot open /net/udp/new");
    };
    let mut buf = alloc::vec![0u8; 32];
    let n = vfs_read(nfd, &mut buf).unwrap_or(0);
    let _ = vfs_close(nfd);
    buf.truncate(n);
    let id: u32 = String::from_utf8_lossy(&buf)
        .trim()
        .parse()
        .map_err(|_| "bad socket id")?;

    let ctl_path = alloc::format!("/net/udp/{}/ctl", id);
    let data_path = alloc::format!("/net/udp/{}/data", id);

    // Connect to DNS server
    {
        let Ok(fd) = vfs_open(&ctl_path, O_WRONLY) else {
            return Err("cannot open udp ctl");
        };
        let cmd = alloc::format!("connect {} 53", dns_server);
        let _ = vfs_write(fd, cmd.as_bytes());
        let _ = vfs_close(fd);
    }

    // Send DNS query
    let query = build_dns_query(name);
    {
        let Ok(fd) = vfs_open(&data_path, O_WRONLY) else {
            return Err("cannot open udp data");
        };
        let mut pkt = alloc::vec![0u8; 4 + query.len()];
        pkt[..4].copy_from_slice(&(query.len() as u32).to_le_bytes());
        pkt[4..].copy_from_slice(&query);
        let _ = vfs_write(fd, &pkt);
        let _ = vfs_close(fd);
    }

    // Wait for response
    let deadline = stem::time::now() + stem::time::Duration::from_millis(3000);
    loop {
        let Ok(fd) = vfs_open(&data_path, O_RDONLY) else {
            return Err("cannot open udp data for read");
        };
        let mut resp = alloc::vec![0u8; 2048 + 4];
        let n = vfs_read(fd, &mut resp).unwrap_or(0);
        let _ = vfs_close(fd);
        if n >= 5 {
            if let Some(addr) = parse_first_a(&resp[4..n]) {
                return Ok(alloc::format!("{}.{}.{}.{}", addr[0], addr[1], addr[2], addr[3]));
            }
            return Err("no A record");
        }
        if stem::time::now() >= deadline {
            return Err("DNS timeout");
        }
        stem::time::sleep_ms(50);
    }
}

/// Attempt a TCP connection to `ip:port`, return round-trip time in ms or error.
fn tcp_probe(ip: &str, port: u16) -> Result<u64, &'static str> {
    // Allocate TCP socket
    let id = {
        let Ok(fd) = vfs_open("/net/tcp/new", O_RDONLY) else {
            return Err("cannot open /net/tcp/new");
        };
        let mut buf = alloc::vec![0u8; 32];
        let n = vfs_read(fd, &mut buf).unwrap_or(0);
        let _ = vfs_close(fd);
        buf.truncate(n);
        let s = String::from_utf8_lossy(&buf).trim().to_string();
        let id: u32 = s.parse().map_err(|_| "bad socket id")?;
        id
    };

    let ctl_path = alloc::format!("/net/tcp/{}/ctl", id);
    let status_path = alloc::format!("/net/tcp/{}/status", id);

    // Initiate connection
    let t0 = stem::time::now();
    {
        let Ok(fd) = vfs_open(&ctl_path, O_WRONLY) else {
            return Err("cannot open tcp ctl");
        };
        let cmd = alloc::format!("connect {} {}", ip, port);
        let _ = vfs_write(fd, cmd.as_bytes());
        let _ = vfs_close(fd);
    }

    // Poll status until established, closed, or timeout
    let timeout = stem::time::Duration::from_millis(5000);
    loop {
        let status = read_file(&status_path);
        if status.contains("established") {
            let elapsed = (stem::time::now() - t0).as_millis();
            return Ok(elapsed);
        }
        if status.contains("closed") || status.contains("error") {
            return Err("connection refused");
        }
        if (stem::time::now() - t0) >= timeout {
            return Err("timeout");
        }
        stem::time::sleep_ms(20);
    }
}

#[stem::main]
fn main(_arg: usize) -> ! {
    let args = get_args();

    // Parse arguments: ping [-c count] [-p port] <host>
    let mut count: u32 = 4;
    let mut port: u16 = 80;
    let mut host = String::new();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-c" => {
                i += 1;
                if i < args.len() {
                    count = args[i].parse().unwrap_or(4);
                }
            }
            "-p" => {
                i += 1;
                if i < args.len() {
                    port = args[i].parse().unwrap_or(80);
                }
            }
            _ => {
                host = args[i].clone();
            }
        }
        i += 1;
    }

    if host.is_empty() {
        print(2, "usage: ping [-c count] [-p port] <host>\n");
        stem::syscall::exit(1);
    }

    // Resolve hostname
    let ip = match resolve(&host) {
        Ok(ip) => ip,
        Err(e) => {
            let msg = alloc::format!("ping: {}: {}\n", host, e);
            print(2, &msg);
            stem::syscall::exit(1);
        }
    };

    let header = alloc::format!(
        "PING {} ({}) port {} (TCP)\n",
        host, ip, port
    );
    print(1, &header);

    let mut transmitted = 0u32;
    let mut received = 0u32;
    let mut total_ms: u64 = 0;
    let mut min_ms: u64 = u64::MAX;
    let mut max_ms: u64 = 0;

    for seq in 1..=count {
        match tcp_probe(&ip, port) {
            Ok(ms) => {
                received += 1;
                total_ms += ms;
                if ms < min_ms {
                    min_ms = ms;
                }
                if ms > max_ms {
                    max_ms = ms;
                }
                let msg = alloc::format!(
                    "tcp_seq={} host={} port={} time={}ms\n",
                    seq, ip, port, ms
                );
                print(1, &msg);
            }
            Err(e) => {
                let msg = alloc::format!("tcp_seq={} host={} port={} {}\n", seq, ip, port, e);
                print(1, &msg);
            }
        }
        transmitted += 1;
        if seq < count {
            stem::time::sleep_ms(1000);
        }
    }

    let loss = if transmitted > 0 {
        100 * (transmitted - received) / transmitted
    } else {
        100
    };

    let summary = alloc::format!(
        "\n--- {} ping statistics ---\n{} probes transmitted, {} received, {}% lost\n",
        host, transmitted, received, loss
    );
    print(1, &summary);

    if received > 0 {
        let avg = total_ms / received as u64;
        let rtt = alloc::format!(
            "rtt min/avg/max = {}/{}/{} ms\n",
            min_ms, avg, max_ms
        );
        print(1, &rtt);
    }

    stem::syscall::exit(if received > 0 { 0 } else { 1 })
}
