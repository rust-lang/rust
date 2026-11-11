fn main() {
    Some.nonexistent_method(); //~ ERROR: no method named `nonexistent_method` found
    Some.nonexistent_field; //~ ERROR: no field `nonexistent_field`
}
