use core::io::println;

fn main() {
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
    println((copy v[0]).to_str());
    println((copy v[1]).to_str());
    println((copy v[2]).to_str());
    println((copy v[3]).to_str());
    println((copy v[4]).to_str());
}

