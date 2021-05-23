// format_strings is false by default.

println!(
    "DirEntry {{ \
        binary_name: {:<64}, \
        context_id: {:>2}, \
        file_size: {:>6}, \
        offset: 0x {:>08X}, \
        actual_crc: 0x{:>08X} \
    }}",
    dir_entry.binary_name,
    dir_entry.context_id,
    dir_entry.file_size,
    dir_entry.offset,
    dir_entry.actual_crc
);
