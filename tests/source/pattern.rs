fn main() {
    let z = match x {
        "pat1" => 1,
        ( ref  x, ref  mut  y /*comment*/) => 2,
    };

    if let <  T as  Trait   > :: CONST = ident {
        do_smth();
    }

    let Some ( ref   xyz  /*   comment!   */) = opt;

    if let  None  =   opt2 { panic!("oh noes"); }
}
