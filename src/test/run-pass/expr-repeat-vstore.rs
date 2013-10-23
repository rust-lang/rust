#[feature(managed_boxes)];

use std::io::println;

pub fn main() {
    let v: ~[int] = ~[ 1, ..5 ];
    println(v[0].to_str());
    println(v[1].to_str());
    println(v[2].to_str());
    println(v[3].to_str());
    println(v[4].to_str());
    let v: @[int] = @[ 2, ..5 ];
    println(v[0].to_str());
    println(v[1].to_str());
    println(v[2].to_str());
    println(v[3].to_str());
    println(v[4].to_str());
    let v: @mut [int] = @mut [ 3, ..5 ];
    println((v[0]).to_str());
    println((v[1]).to_str());
    println((v[2]).to_str());
    println((v[3]).to_str());
    println((v[4]).to_str());
}
