macro_rules! count {
    ( $( $e:stmt ),* ) => {
        ${ count(e) }
        //~^ ERROR meta-variable expressions are unstable
    };
}

fn main() {
}
