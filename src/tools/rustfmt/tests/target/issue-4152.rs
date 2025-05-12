// rustfmt-hard_tabs: true

macro_rules! bit {
	($bool:expr) => {
		if $bool {
			1;
			1
		} else {
			0;
			0
		}
	};
}
macro_rules! add_one {
	($vec:expr) => {{
		$vec.push(1);
	}};
}
