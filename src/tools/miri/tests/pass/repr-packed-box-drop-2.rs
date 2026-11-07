struct HasDrop;

impl Drop for HasDrop {
    fn drop(&mut self) {}
}

#[repr(C, packed)]
struct P(u8, Box<HasDrop>);

fn main() {
    let p = P(0, Box::new(HasDrop));
    let _h = *p.1;
}
