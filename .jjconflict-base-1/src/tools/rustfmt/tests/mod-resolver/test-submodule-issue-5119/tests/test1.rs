mod test1 {
#[cfg(unix)]
mod sub1;
#[cfg(not(unix))]
mod sub2;

mod sub3;
}
