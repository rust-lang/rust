//@ dont-require-annotations: NOTE

struct Bar; //~ NOTE previous definition of the type `Bar` here
struct Bar; //~ ERROR E0428

fn main () {
}
