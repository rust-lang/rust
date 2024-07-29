#[test]
fn quickack() {
    use crate::net::test::next_test_ip4;
    use crate::net::{TcpListener, TcpStream};
    use crate::os::net::linux_ext::tcp::TcpStreamExt;

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

#[test]
#[cfg(target_os = "linux")]
fn deferaccept() {
    use crate::net::test::next_test_ip4;
    use crate::net::{TcpListener, TcpStream};
    use crate::os::net::linux_ext::tcp::TcpStreamExt;

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
    stream.set_deferaccept(1).expect("set_deferaccept failed");
    assert_eq!(stream.deferaccept().unwrap(), 1);
    stream.set_deferaccept(0).expect("set_deferaccept failed");
    assert_eq!(stream.deferaccept().unwrap(), 0);
}
