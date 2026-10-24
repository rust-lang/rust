//@ check-pass

macro_rules! m {
    (static $name:ident: $t:ty = $e:expr) => {
        let $name: $t = $e;
    }
}

fn main() {
    m! {
        // Tricky: the trailing `>>=` token here is broken twice:
        // - into `>` and `>=`
        // - then the `>=` is broken into `>` and `=`
        static _x: Vec<Vec<u32>>= vec![]
    }
}
