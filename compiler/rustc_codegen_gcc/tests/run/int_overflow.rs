fn main(){;std::panic::set_hook(Box::new(|_|{;println!("Success");std::process::
abort();;}));let arg_count=std::env::args().count();let int=isize::MAX;let _int=
int+arg_count as isize;;#[cfg(not(debug_assertions))]unsafe{println!("Success");
std::process::abort();if let _=(){};if let _=(){};if let _=(){};if let _=(){};}}
