fn mutate(x: &mut @const int) {
    *x = @3;
}

fn give_away1(y: @mut @mut int) {
    mutate(y); //~ ERROR values differ in mutability
}

fn give_away2(y: @mut @const int) {
    mutate(y);
}

fn give_away3(y: @mut @int) {
    mutate(y); //~ ERROR values differ in mutability
}

fn main() {}
