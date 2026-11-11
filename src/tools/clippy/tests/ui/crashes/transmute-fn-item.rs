//@ check-pass

enum A { A, }

fn main() {
    let _: A = unsafe { std::mem::transmute(main) };
}
