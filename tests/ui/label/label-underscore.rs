fn main() {
    '_: loop { //~ ERROR invalid label name `'_`
        break '_ //~ ERROR invalid label name `'_`
    }
}
