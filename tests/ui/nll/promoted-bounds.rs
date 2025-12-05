fn shorten_lifetime<'a, 'b, 'min>(a: &'a i32, b: &'b i32) -> &'min i32
where
    'a: 'min,
    'b: 'min,
{
    if *a < *b {
        &a
    } else {
        &b
    }
}

fn main() {
    let promoted_fn_item_ref = &shorten_lifetime;

    let a = &5;
    let ptr = {
        let l = 3;
        let b = &l; //~ ERROR does not live long enough
        let c = promoted_fn_item_ref(a, b);
        c
    };

    println!("ptr = {:?}", ptr);
}
