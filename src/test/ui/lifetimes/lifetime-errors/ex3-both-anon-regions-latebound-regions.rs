fn foo<'a,'b>(x: &mut Vec<&'a u8>, y: &'b u8) {
    x.push(y); //~ ERROR lifetime mismatch
}

fn main() { }
