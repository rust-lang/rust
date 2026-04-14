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
    let mut buf = alloc::vec![0u8; 64];
    let n = vfs_read(fd, &mut buf).unwrap_or(0);
    let _ = vfs_close(fd);
    buf.truncate(n);
    String::from_utf8_lossy(&buf).trim().into()
}

/// Build a DNS A-record query packet for the given name.
fn build_dns_query(name: &str) -> Vec<u8> {
    let mut pkt: Vec<u8> = Vec::new();
    // Transaction ID
    pkt.extend_from_slice(&[0x12, 0x34]);
    // Flags: standard query, recursion desired
    pkt.extend_from_slice(&[0x01, 0x00]);
    // Questions: 1
    pkt.extend_from_slice(&[0x00, 0x01]);
    // Answer/Auth/Additional RRs: 0
    pkt.extend_from_slice(&[0x00, 0x00, 0x00, 0x00, 0x00, 0x00]);
    // Encode QNAME
    for label in name.split('.') {
        pkt.push(label.len() as u8);
        pkt.extend_from_slice(label.as_bytes());
    }
    pkt.push(0); // root label
    // QTYPE A (1), QCLASS IN (1)
    pkt.extend_from_slice(&[0x00, 0x01, 0x00, 0x01]);
    pkt
}

/// Parse an A-record answer from a DNS response.  Returns up to 8 addresses.
fn parse_dns_response(data: &[u8]) -> Vec<[u8; 4]> {
    let mut results = Vec::new();
    if data.len() < 12 {
        return results;
    }
    let ancount = u16::from_be_bytes([data[6], data[7]]) as usize;
    if ancount == 0 {
        return results;
    }

    // Skip the question section by scanning past the QNAME + 4 bytes.
    let mut pos = 12;
    // Skip QNAME in question
    loop {
        if pos >= data.len() {
            return results;
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
    pos += 4; // skip QTYPE + QCLASS

    for _ in 0..ancount {
        if pos >= data.len() {
            break;
        }
        // Skip NAME (may be a pointer)
        let b = data[pos];
        if b & 0xC0 == 0xC0 {
            pos += 2;
        } else {
            loop {
                if pos >= data.len() {
                    return results;
                }
                let l = data[pos] as usize;
                pos += 1;
                if l == 0 {
                    break;
                }
                pos += l;
            }
        }
        // TYPE (2) + CLASS (2) + TTL (4) + RDLENGTH (2)
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
            // A record
            results.push([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
        }
        pos += rdlen;
    }
    results
}

/// Look up A records for `name` using DNS server `server_ip` (dotted-decimal).
/// Returns the resolved addresses or an error message.
fn lookup(name: &str, server_ip: &str) -> Result<Vec<[u8; 4]>, &'static str> {
    // Allocate a UDP socket via /net/udp/new
    let new_id = {
        let Ok(fd) = vfs_open("/net/udp/new", O_RDONLY) else {
            return Err("cannot open /net/udp/new");
        };
        let mut buf = alloc::vec![0u8; 32];
        let n = vfs_read(fd, &mut buf).unwrap_or(0);
        let _ = vfs_close(fd);
        buf.truncate(n);
        let s = String::from_utf8_lossy(&buf).trim().to_string();
        let id: u32 = s.parse().map_err(|_| "bad socket id")?;
        id
    };

    let ctl_path = alloc::format!("/net/udp/{}/ctl", new_id);
    let data_path = alloc::format!("/net/udp/{}/data", new_id);

    // Connect the socket to the DNS server on port 53
    {
        let Ok(fd) = vfs_open(&ctl_path, O_WRONLY) else {
            return Err("cannot open udp ctl");
        };
        let cmd = alloc::format!("connect {} 53", server_ip);
        let _ = vfs_write(fd, cmd.as_bytes());
        let _ = vfs_close(fd);
    }

    // Build and send the DNS query (length-prefixed)
    let query = build_dns_query(name);
    {
        let Ok(fd) = vfs_open(&data_path, O_WRONLY) else {
            return Err("cannot open udp data for write");
        };
        let mut pkt = alloc::vec![0u8; 4 + query.len()];
        let len32 = query.len() as u32;
        pkt[..4].copy_from_slice(&len32.to_le_bytes());
        pkt[4..].copy_from_slice(&query);
        let _ = vfs_write(fd, &pkt);
        let _ = vfs_close(fd);
    }

    // Poll for the response (up to ~3 seconds, spinning)
    let deadline = stem::time::now() + stem::time::Duration::from_millis(3000);
    loop {
        let Ok(fd) = vfs_open(&data_path, O_RDONLY) else {
            return Err("cannot open udp data for read");
        };
        let mut buf = alloc::vec![0u8; 2048 + 4];
        let n = vfs_read(fd, &mut buf).unwrap_or(0);
        let _ = vfs_close(fd);

        if n >= 5 {
            // First 4 bytes are the length prefix
            let payload = &buf[4..n];
            let addrs = parse_dns_response(payload);
            return Ok(addrs);
        }

        if stem::time::now() >= deadline {
            return Err("timeout");
        }
        stem::time::sleep_ms(50);
    }
}

#[stem::main]
fn main(_arg: usize) -> ! {
    let args = get_args();

    if args.len() < 2 {
        print(2, "usage: nslookup <hostname> [server]\n");
        stem::syscall::exit(1);
    }

    let hostname = &args[1];

    // Determine DNS server: use argument if provided, else read from /net/dns/server
    let server = if args.len() >= 3 {
        args[2].clone()
    } else {
        let s = read_file("/net/dns/server");
        if s.is_empty() || s == "0.0.0.0" {
            // Fallback to well-known public resolver
            String::from("8.8.8.8")
        } else {
            s
        }
    };

    let out = alloc::format!("Server:\t{}\n\n", server);
    print(1, &out);

    match lookup(hostname, &server) {
        Ok(addrs) if !addrs.is_empty() => {
            let out = alloc::format!("Name:\t{}\n", hostname);
            print(1, &out);
            for addr in &addrs {
                let out =
                    alloc::format!("Address: {}.{}.{}.{}\n", addr[0], addr[1], addr[2], addr[3]);
                print(1, &out);
            }
        }
        Ok(_) => {
            let out = alloc::format!("nslookup: {}: no records found\n", hostname);
            print(2, &out);
            stem::syscall::exit(1);
        }
        Err(e) => {
            let out = alloc::format!("nslookup: {}: {}\n", hostname, e);
            print(2, &out);
            stem::syscall::exit(1);
        }
    }

    stem::syscall::exit(0)
}
