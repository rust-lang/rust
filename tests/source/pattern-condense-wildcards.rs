// rustfmt-condense_wildcard_suffices: true

fn main() {
    match x {
        Butt (_,_) => "hah",
        Tup (_) =>  "nah",
        Quad (_,_, x,_) =>   " also no rewrite",
        Quad (x, _, _, _) => "condense me pls",
        Weird (x, _, _, /* dont condense before */ _, _, _) => "pls work",
    }
}
