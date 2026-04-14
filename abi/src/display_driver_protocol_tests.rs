#[cfg(test)]
mod tests {
    extern crate std;

    use crate::display_driver_protocol as drvproto;
    use crate::driver_frame::FrameReader;

    #[test]
    fn round_trip_payloads_and_golden_bytes() {
        let hello = drvproto::HelloPayload {
            proto_major: 1,
            proto_minor: 2,
            want_caps: 0x0D0C0B0A,
        };
        let mut hello_bytes = [0u8; drvproto::HELLO_PAYLOAD_WIRE_SIZE];
        let len = drvproto::encode_hello_payload_le(&hello, &mut hello_bytes).unwrap();
        assert_eq!(len, drvproto::HELLO_PAYLOAD_WIRE_SIZE);
        assert_eq!(
            hello_bytes,
            [0x01, 0x00, 0x02, 0x00, 0x0A, 0x0B, 0x0C, 0x0D]
        );
        let decoded_hello = drvproto::decode_hello_payload_le(&hello_bytes).unwrap();
        assert_eq!(decoded_hello.proto_major, 1);
        assert_eq!(decoded_hello.proto_minor, 2);
        assert_eq!(decoded_hello.want_caps, 0x0D0C0B0A);

        let welcome = drvproto::WelcomePayload {
            proto_major: 1,
            proto_minor: 0,
            have_caps: 0x44332211,
            max_rects: 0x55AA,
            reserved: 0x0001,
        };
        let mut welcome_bytes = [0u8; drvproto::WELCOME_PAYLOAD_WIRE_SIZE];
        let len = drvproto::encode_welcome_payload_le(&welcome, &mut welcome_bytes).unwrap();
        assert_eq!(len, drvproto::WELCOME_PAYLOAD_WIRE_SIZE);
        assert_eq!(
            welcome_bytes,
            [0x01, 0x00, 0x00, 0x00, 0x11, 0x22, 0x33, 0x44, 0xAA, 0x55, 0x01, 0x00]
        );
        let decoded_welcome = drvproto::decode_welcome_payload_le(&welcome_bytes).unwrap();
        assert_eq!(decoded_welcome.proto_major, 1);
        assert_eq!(decoded_welcome.proto_minor, 0);
        assert_eq!(decoded_welcome.have_caps, 0x44332211);
        assert_eq!(decoded_welcome.max_rects, 0x55AA);
        assert_eq!(decoded_welcome.reserved, 0x0001);

        let bind = drvproto::BindPayload {
            fb_fd: 0x55667788,
            _pad: 0x11223344,
            width: 0xAABBCCDD,
            height: 0x01020304,
            stride: 0x05060708,
            format: 0x0A0B0C0D,
        };
        let mut bind_bytes = [0u8; drvproto::BIND_PAYLOAD_WIRE_SIZE];
        let len = drvproto::encode_bind_payload_le(&bind, &mut bind_bytes).unwrap();
        assert_eq!(len, drvproto::BIND_PAYLOAD_WIRE_SIZE);
        assert_eq!(
            bind_bytes,
            [
                0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11, 0xDD, 0xCC, 0xBB, 0xAA, 0x04, 0x03,
                0x02, 0x01, 0x08, 0x07, 0x06, 0x05, 0x0D, 0x0C, 0x0B, 0x0A,
            ]
        );
        let decoded_bind = drvproto::decode_bind_payload_le(&bind_bytes).unwrap();
        assert_eq!(decoded_bind.fb_fd, 0x55667788);
        assert_eq!(decoded_bind.width, 0xAABBCCDD);
        assert_eq!(decoded_bind.height, 0x01020304);
        assert_eq!(decoded_bind.stride, 0x05060708);
        assert_eq!(decoded_bind.format, 0x0A0B0C0D);

        let mut present_header = [0u8; drvproto::PRESENT_HEADER_WIRE_SIZE];
        drvproto::encode_present_header_with_flags_le(
            2,
            drvproto::PRESENT_FLAG_FULLFRAME,
            &mut present_header,
        )
        .unwrap();
        assert_eq!(
            present_header,
            [0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00]
        );
        let decoded_header = drvproto::decode_present_header_le(&present_header).unwrap();
        assert_eq!(decoded_header.rect_count, 2);
        assert_eq!(decoded_header._pad, drvproto::PRESENT_FLAG_FULLFRAME);

        let rects = [
            drvproto::Rect {
                x: 1,
                y: 2,
                w: 3,
                h: 4,
            },
            drvproto::Rect {
                x: 5,
                y: 6,
                w: 7,
                h: 8,
            },
        ];
        let mut present_payload = [0u8; 8 + 2 * 16];
        let len = drvproto::encode_present_payload_with_flags_le(
            2,
            0,
            rects.into_iter(),
            &mut present_payload,
        )
        .unwrap();
        assert_eq!(len, present_payload.len());
        assert_eq!(
            present_payload[0..8],
            [0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
        );
        assert_eq!(
            present_payload[8..24],
            [1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0]
        );
        assert_eq!(
            present_payload[24..40],
            [5, 0, 0, 0, 6, 0, 0, 0, 7, 0, 0, 0, 8, 0, 0, 0]
        );

        let mut msg = [0u8; 64];
        let msg_len =
            drvproto::encode_message(&mut msg, drvproto::MSG_HELLO, &hello_bytes).unwrap();
        assert_eq!(&msg[0..4], &drvproto::DRIVER_MAGIC.to_le_bytes());
        assert_eq!(&msg[4..6], &drvproto::DRIVER_VERSION.to_le_bytes());
        assert_eq!(&msg[6..8], &drvproto::MSG_HELLO.to_le_bytes());
        assert_eq!(&msg[8..12], &(hello_bytes.len() as u32).to_le_bytes());
        assert_eq!(&msg[12..12 + hello_bytes.len()], &hello_bytes);
        assert_eq!(msg_len, drvproto::HEADER_SIZE + hello_bytes.len());
    }

    #[test]
    fn frame_reader_partial_reads_and_resync() {
        let hello = drvproto::HelloPayload {
            proto_major: 1,
            proto_minor: 0,
            want_caps: 0x00000003,
        };
        let mut hello_bytes = [0u8; drvproto::HELLO_PAYLOAD_WIRE_SIZE];
        drvproto::encode_hello_payload_le(&hello, &mut hello_bytes).unwrap();
        let mut msg = [0u8; 64];
        let msg_len =
            drvproto::encode_message(&mut msg, drvproto::MSG_HELLO, &hello_bytes).unwrap();
        let msg = &msg[..msg_len];

        let mut reader = FrameReader::<128>::new();
        for (i, byte) in msg.iter().enumerate() {
            reader.push(&[*byte]);
            let got = reader.next_message();
            if i + 1 == msg.len() {
                assert!(got.is_some());
            } else {
                assert!(got.is_none());
            }
        }

        let mut reader = FrameReader::<256>::new();
        let mut msg2 = [0u8; 64];
        let msg2_len = drvproto::encode_message(&mut msg2, drvproto::MSG_BIND, &[]).unwrap();
        let mut stream = std::vec::Vec::new();
        stream.extend_from_slice(msg);
        stream.push(0xAA);
        stream.extend_from_slice(&msg2[..msg2_len]);
        reader.push(&stream);
        let first = reader.next_message().unwrap();
        assert_eq!(first.0.msg_type, drvproto::MSG_HELLO);
        let second = reader.next_message().unwrap();
        assert_eq!(second.0.msg_type, drvproto::MSG_BIND);
    }

    #[test]
    fn frame_reader_back_to_back_messages() {
        let mut msg1 = [0u8; 64];
        let mut msg2 = [0u8; 64];
        let msg1_len = drvproto::encode_message(&mut msg1, drvproto::MSG_ACK, &[]).unwrap();
        let msg2_len =
            drvproto::encode_message(&mut msg2, drvproto::MSG_ERR, &[0, 0, 0, 0]).unwrap();

        let mut buf = std::vec::Vec::new();
        buf.extend_from_slice(&msg1[..msg1_len]);
        buf.extend_from_slice(&msg2[..msg2_len]);

        let mut reader = FrameReader::<256>::new();
        reader.push(&buf);
        let first = reader.next_message().unwrap();
        assert_eq!(first.0.msg_type, drvproto::MSG_ACK);
        let second = reader.next_message().unwrap();
        assert_eq!(second.0.msg_type, drvproto::MSG_ERR);
    }
}
