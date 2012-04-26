#[doc="
High-level interface to libuv's TCP functionality
"];

#[cfg(ignore)]
mod test {
    #[test]
    fn test_gl_tcp_ipv4_request() {
        let ip = "127.0.0.1";
        let port = 80u;
        let expected_read_msg = "foo";
        let actual_write_msg = "bar";
        let addr = ipv4::address(ip, port);

        let data_po = comm::port::<[u8]>();
        let data_ch = comm::chan(data_po);
        
        alt connect(addr) {
          tcp_connected(tcp_stream) {
            let write_data = str::as_buf(actual_write_msg);
            alt write(tcp_stream, [write_data]) {
              tcp_write_success {
                let mut total_read_data: [u8] = []
                let reader_po = read_start(tcp_stream);
                loop {
                    alt comm::recv(reader_po) {
                      new_read_data(data) {
                        total_read_data += data;
                        // theoretically, we could keep iterating, here, if
                        // we expect the server on the other end to keep
                        // streaming/chunking data to us, but..
                        read_stop(tcp_stream);
                        break;
                      }
                      done_reading {
                        break;
                      }
                      error {
                        fail "erroring occured during read attempt.. FIXME need info";
                      }
                    }
                }
                comm::send(data_ch, total_read_data);
              }
              tcp_write_error {
                fail "error during write attempt.. FIXME need err info";
              }
            }
          }
          tcp_connect_error {
            fail "error during connection attempt.. FIXME need err info..";
          }
        }

        let actual_data = comm::recv(data_po);
        let resp = str::from_bytes(actual_data);
        log(debug, "DATA RECEIVED: "+resp);
    }
}