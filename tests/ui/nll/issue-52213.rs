fn transmute_lifetime<'a, 'b, T>(t: &'a (T,)) -> &'b T {
    match (&t,) {
        ((u,),) => u,
        //~^ ERROR lifetime may not live long enough
    }
}

fn main() {
    let x = {
        let y = Box::new((42,));
        transmute_lifetime(&y)
    };

    println!("{}", x);
}
