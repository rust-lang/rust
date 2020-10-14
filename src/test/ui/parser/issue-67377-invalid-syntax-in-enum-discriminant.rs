mod a {
    use std::marker::PhantomData;

    enum Bug {
        V = [PhantomData; { [ () ].len() ].len() as isize,
        //~^ ERROR mismatched closing delimiter: `]`
        //~| ERROR mismatched closing delimiter: `]`
        //~| ERROR mismatched closing delimiter: `]`
        //~| ERROR mismatched closing delimiter: `]`
    }
}

mod b {
    enum Bug {
        V = [Vec::new; { [].len()  ].len() as isize,
        //~^ ERROR mismatched closing delimiter: `]`
        //~| ERROR mismatched closing delimiter: `]`
        //~| ERROR mismatched closing delimiter: `]`
        //~| ERROR mismatched closing delimiter: `]`
    }
}

mod c {
    enum Bug {
        V = [Vec::new; { [0].len() ].len() as isize,
        //~^ ERROR mismatched closing delimiter: `]`
        //~| ERROR mismatched closing delimiter: `]`
        //~| ERROR mismatched closing delimiter: `]`
        //~| ERROR mismatched closing delimiter: `]`
    }
}

fn main() {}
