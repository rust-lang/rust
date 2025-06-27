static buf: &mut [u8] = &mut [1u8,2,3,4,5,7];   //~ ERROR mutable borrows of temporaries
fn write<T: AsRef<[u8]>>(buffer: T) { }

fn main() {
    write(&buf);
    buf[0]=2;                                   //~ ERROR E0594
}
