// The error message here still is pretty confusing.

fn main() {
    let mut result = vec![1];
    // The type of `result` is constrained to be `Vec<{integer}>` here.
    // But the logic we use to find what expression constrains a type
    // is not sophisticated enough to know this.

    let mut vector = Vec::new();
    vector.sort();
    result.push(vector);
    //~^ ERROR mismatched types
    // So it thinks that the type of `result` is constrained here.
}

fn example2() {
    let mut x = vec![1];
    x.push("");
    //~^ ERROR mismatched types
}
