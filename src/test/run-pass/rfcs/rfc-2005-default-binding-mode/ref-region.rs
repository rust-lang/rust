// run-pass
fn foo<'a, 'b>(x: &'a &'b Option<u32>) -> &'a u32 {
    let x: &'a &'a Option<u32> = x;
    match x {
        Some(r) => {
            let _: &u32 = r;
            r
        },
        &None => panic!(),
    }
}

pub fn main() {
    let x = Some(5);
    foo(&&x);
}
