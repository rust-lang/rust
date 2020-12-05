fn main() {
    format!(r"{}{}{}", named_arg=1); //~ ERROR invalid reference to positional arguments 1 and 2
}
