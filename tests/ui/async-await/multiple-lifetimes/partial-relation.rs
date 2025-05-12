//@ edition:2018
//@ run-pass

async fn lotsa_lifetimes<'a, 'b, 'c>(a: &'a u32, b: &'b u32, c: &'c u32) -> (&'a u32, &'b u32)
    where 'b: 'a
{
    let _ = (a, c);
    (b, b)
}

fn main() {
    let _ = lotsa_lifetimes(&22, &44, &66);
}
