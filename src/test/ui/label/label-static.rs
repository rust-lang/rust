fn main() {
    'static: loop { //~ ERROR invalid label name `'static`
        break 'static //~ ERROR invalid label name `'static`
    }
}
