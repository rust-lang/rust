//~ ERROR overflow evaluating the requirement `{closure@$DIR/overflow-during-mono.rs:13:41: 13:44}: Sized`
//@ build-fail

#![recursion_limit = "32"]

fn quicksort<It: Clone + Iterator<Item = T>, I: IntoIterator<IntoIter = It>, T: Ord>(
    i: I,
) -> Vec<T> {
    let mut i = i.into_iter();

    match i.next() {
        Some(x) => {
            let less = i.clone().filter(|y| y < &x);
            let greater = i.filter(|y| &x <= y);

            let mut v = quicksort(less);
            let u = quicksort(greater);
            v.push(x);
            v.extend(u);
            v
        }
        None => vec![],
    }
}

fn main() {
    println!("{:?}", quicksort([5i32, 1, 6, 3, 6, 1, 9, 0, -1, 6, 8]));
}
