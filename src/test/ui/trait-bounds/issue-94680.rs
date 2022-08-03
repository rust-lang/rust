// check-pass

fn main() {
    println!("{:?}", {
        type T = ();

        pub fn cloneit(it: &'_ mut T) -> (&'_ mut T, &'_ mut T)
        where
            for<'any> &'any mut T: Clone,
            //~^ WARNING the where-clause bound
        {
            (it.clone(), it)
        }
    });
}
