import core::*;

use std;
import std::net;

#[test]
fn test_format_ip() {
    assert (net::format_addr(net::ipv4(127u8, 0u8, 0u8, 1u8)) == "127.0.0.1")
}

#[test]
fn test_parse_ip() {
    assert (net::parse_addr("127.0.0.1") == net::ipv4(127u8, 0u8, 0u8, 1u8));
}
