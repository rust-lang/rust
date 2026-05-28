#[test]
fn quickack() {
    use crate::net::test::LOCALHOST_IP4;
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

    let listener = t!(TcpListener::bind(LOCALHOST_IP4));
    let addr = t!(listener.local_addr());

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
    use crate::net::test::LOCALHOST_IP4;
    use crate::net::{TcpListener, TcpStream};
    use crate::os::net::linux_ext::tcp::TcpStreamExt;
    use crate::time::Duration;

    macro_rules! t {
        ($e:expr) => {
            match $e {
                Ok(t) => t,
                Err(e) => panic!("received error for `{}`: {}", stringify!($e), e),
            }
        };
    }

    let one = Duration::from_secs(1u64);
    let zero = Duration::from_secs(0u64);

    let listener = t!(TcpListener::bind(LOCALHOST_IP4));
    let addr = t!(listener.local_addr());
    let stream = t!(TcpStream::connect(&("localhost", addr.port())));
    stream.set_deferaccept(one).expect("set_deferaccept failed");
    assert_eq!(stream.deferaccept().unwrap(), one);
    stream.set_deferaccept(zero).expect("set_deferaccept failed");
    assert_eq!(stream.deferaccept().unwrap(), zero);
}
