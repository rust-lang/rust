//@ known-bug: #122823
//@ compile-flags: -Copt-level=0
// ignore-tidy-linelength

use std::vec::Vec;
use std::iter::Peekable;

pub fn main() {
    let packet = decode(vec![1,0,1,0]);
}

pub fn decode(bitstream: Vec<u64>) -> Packet {
    let mut bitstream_itr = bitstream.into_iter().peekable();
    return match decode_packet(&mut bitstream_itr) {
        Some(p) => p,
        None    => panic!("expected outer packet"),
    }
}

pub fn decode_packets<I: Iterator<Item = u64>>(itr: &mut Peekable<I>) -> Vec<Packet> {
    let mut res = Vec::new();
    loop {
        match decode_packet(itr) {
            Some(p) => { res.push(p); },
            None    => break
        }
    }

    return res;
}

pub fn decode_packet<I: Iterator<Item = u64>>(itr: &mut Peekable<I>) -> Option<Packet> {
    // get version digits
    let version = extend_number(0, itr, 3)?;
    let type_id = extend_number(0, itr, 3)?;
    return operator_packet(version, type_id, itr);
}

pub fn operator_packet<I: Iterator<Item = u64>>(version: u64, type_id: u64, itr: &mut Peekable<I>) -> Option<Packet> {
    let p = OperatorPacket {
        version: version,
        type_id: type_id,
        packets: decode_packets(&mut itr.take(0).peekable()),
    };

    return Some(Packet::Operator(p));
}

pub fn extend_number<I: Iterator<Item = u64>>(num: u64, itr: &mut Peekable<I>, take: u64) -> Option<u64> {
    let mut value = num;
    for _ in 0..take {
        value *= 2;
        value += itr.next()?;
    }

    return Some(value);
}

#[derive(Debug)]
pub enum Packet {
    Operator(OperatorPacket),
}

#[derive(Debug)]
pub struct OperatorPacket {
    version: u64,
    type_id: u64,
    packets: Vec<Packet>
}
