fn stati<T: 'stati>() {}
//~^ ERROR use of undeclared lifetime name `'stati`

fn statoc<T: 'statoc>() {}
//~^ ERROR use of undeclared lifetime name `'statoc`

fn main() {}
