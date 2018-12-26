// Make sure that label for continue and break is spanned correctly

fn main() {
    loop {
        continue
        'b //~ ERROR use of undeclared label
        ;
        break
        'c //~ ERROR use of undeclared label
        ;
    }
}
