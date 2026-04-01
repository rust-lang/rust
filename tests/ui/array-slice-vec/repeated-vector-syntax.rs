//@ run-pass

pub fn main() {
    let x = [ [true]; 512 ];
    let y = [ 0; 1 ];

    print!("[");
    for xi in &x[..] {
        print!("{:?}, ", &xi[..]);
    }
    println!("]");
    println!("{:?}", &y[..]);
}
