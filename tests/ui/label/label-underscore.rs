fn main() {
    '_: loop { //~ ERROR labels cannot use keyword names
        break '_ //~ ERROR labels cannot use keyword names
    }
}
