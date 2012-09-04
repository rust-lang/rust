// This test should behave exactly like issue-2735-3
struct defer {
    let b: &mut bool;
    new(b: &mut bool) {
        self.b = b;
    }   
    drop { *(self.b) = true; }
}

fn main() {
    let mut dtor_ran = false;
    let _  = defer(&mut dtor_ran);
    assert(dtor_ran);
}
