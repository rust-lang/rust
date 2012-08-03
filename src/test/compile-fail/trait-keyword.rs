iface foo { } //~ WARN `iface` is deprecated; use `trait`

fn main() { 
    x //~ ERROR unresolved name: x
}