struct S {
    z: f64
}

fn main() {
    let x: f32 = 4.0;
    io::println(x.to_str());
    let y: float = 64.0;
    io::println(y.to_str());
    let z = S { z: 1.0 };
    io::println(z.z.to_str());
}

