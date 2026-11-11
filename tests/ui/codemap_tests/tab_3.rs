// ignore-tidy-file-tab

fn main() {
	let some_vec = vec!["hi"];
	some_vec.into_iter();
	{
		println!("{:?}", some_vec); //~ ERROR borrow of moved
	}
}
