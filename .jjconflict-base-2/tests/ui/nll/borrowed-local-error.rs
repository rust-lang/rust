fn gimme(x: &(u32,)) -> &u32 {
    &x.0
}

fn main() {
    let x = gimme({
        let v = (22,);
        &v
        //~^ ERROR `v` does not live long enough [E0597]
    });
    println!("{:?}", x);
}
