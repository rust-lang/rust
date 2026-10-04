fn main() {
    format!(r"{}{}{}", named_arg=1); //~ ERROR 3 positional arguments in format string, but there is 1 argument
}
