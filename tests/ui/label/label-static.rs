fn main() {
    'static: loop { //~ ERROR labels cannot use keyword names
        break 'static //~ ERROR labels cannot use keyword names
    }
}
