use zoo::fly; //~ ERROR failed to resolve import

mod zoo {
    priv type fly = ();
    priv fn fly() {}
}


fn main() {}
