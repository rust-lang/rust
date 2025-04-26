//@ check-pass

enum E { V() }

fn main() {
    <E>::V() = E::V(); // OK, destructuring assignment
    <E>::V {} = E::V(); // OK, destructuring assignment
}
