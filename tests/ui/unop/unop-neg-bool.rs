fn main() {
    -{ true }; //~ ERROR cannot apply unary operator `-` to type `bool`
    -true; //~ ERROR cannot negate `boolean` literal
}
