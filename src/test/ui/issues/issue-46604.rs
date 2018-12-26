// revisions: ast mir
//[mir]compile-flags: -Z borrowck=mir

static buf: &mut [u8] = &mut [1u8,2,3,4,5,7];   //[ast]~ ERROR E0017
                                                //[mir]~^ ERROR E0017
fn write<T: AsRef<[u8]>>(buffer: T) { }

fn main() {
    write(&buf);
    buf[0]=2;                                   //[ast]~ ERROR E0389
                                                //[mir]~^ ERROR E0594
}
