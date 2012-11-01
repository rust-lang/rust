// xfail-test
fn main() {
    enum State { BadChar, BadSyntax }
    
    match BadChar {
        _ if true => BadChar,
        BadChar | BadSyntax => fail ,
    };
}
