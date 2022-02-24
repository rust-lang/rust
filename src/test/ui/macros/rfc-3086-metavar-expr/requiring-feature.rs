macro_rules! count {
    ( $( $e:stmt ),* ) => {
        ${ count(e) }
        //~^ ERROR meta-variable expressions are unstable
    };
}

macro_rules! a {
    ( $$a:ident ) => {
    //~^ ERROR meta-variable expressions are unstable
    };
}

fn main() {
}
