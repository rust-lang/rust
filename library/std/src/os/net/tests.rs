#[cfg(any(target_os = "android", target_os = "linux",))]
#[test]
fn quickack() {
    use crate::{
        net::{test::next_test_ip4, TcpListener, TcpStream},
        os::net::tcp::TcpStreamExt,
    };

    macro_rules! t {
        ($e:expr) => {
            match $e {
                Ok(t) => t,
                Err(e) => panic!("received error for `{}`: {}", stringify!($e), e),
            }
        };
    }

    let addr = next_test_ip4();
    let _listener = t!(TcpListener::bind(&addr));

    let stream = t!(TcpStream::connect(&("localhost", addr.port())));

    t!(stream.set_quickack(false));
    assert_eq!(false, t!(stream.quickack()));
    t!(stream.set_quickack(true));
    assert_eq!(true, t!(stream.quickack()));
    t!(stream.set_quickack(false));
    assert_eq!(false, t!(stream.quickack()));
}
