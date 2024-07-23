// Fixes: #113379

trait Trait<const S: usize> {}

struct Bug<T>
where
    T: Trait<
        {
            'b: {
                //~^ ERROR [E0308]
                continue 'b; //~ ERROR [E0696]
            }
        },
    >,
{
    t: T,
}

fn f() -> impl Sized {
    'b: {
        continue 'b;
        //~^ ERROR [E0696]
    }
}

fn main() {
    f();
}
