use crate::collections::VecDeque;

#[cfg(not(test))]
impl [u8] {
    #[rustc_allow_incoherent_impl]
    #[unstable(issue = "none", feature = "std_internals")]
    #[allow(dead_code)]
    /// Safety:
    ///    - Must be valid UTF-8
    pub unsafe fn make_utf8_uppercase(&mut self) -> Result<usize, VecDeque<u8>> {
        let mut queue = VecDeque::new();

        let mut read_offset = 0;
        let mut write_offset = 0;

        while let Some((codepoint, width)) =
            unsafe { core::str::next_code_point_with_width(&mut self[read_offset..].iter()) }
        {
            read_offset += width;
            // Queue must be flushed before encode_to_slice_or_else_to_queue is
            // called to ensure proper order of bytes
            dump_queue(&mut queue, &mut self[..read_offset], &mut write_offset);
            let lowercase_char = unsafe { char::from_u32_unchecked(codepoint) };
            for c in lowercase_char.to_uppercase() {
                encode_to_slice_or_else_to_queue(
                    c,
                    &mut queue,
                    &mut self[..read_offset],
                    &mut write_offset,
                );
            }
        }
        assert_eq!(read_offset, self.len());
        if write_offset < read_offset { Ok(write_offset) } else { Err(queue) }
    }

    #[rustc_allow_incoherent_impl]
    #[unstable(issue = "none", feature = "std_internals")]
    #[allow(dead_code)]
    /// Safety:
    ///    - Must be valid UTF-8
    pub unsafe fn make_utf8_lowercase(&mut self) -> Result<usize, VecDeque<u8>> {
        let mut queue = VecDeque::new();

        let mut read_offset = 0;
        let mut write_offset = 0;

        let mut final_sigma_automata = FinalSigmaAutomata::new();
        while let Some((codepoint, width)) =
            unsafe { core::str::next_code_point_with_width(&mut self[read_offset..].iter()) }
        {
            read_offset += width;
            // Queue must be flushed before encode_to_slice_or_else_to_queue is
            // called to ensure proper order of bytes
            dump_queue(&mut queue, &mut self[..read_offset], &mut write_offset);
            let uppercase_char = unsafe { char::from_u32_unchecked(codepoint) };
            if uppercase_char == 'Σ' {
                // Σ maps to σ, except at the end of a word where it maps to ς.
                // See core::str::to_lowercase
                let rest = unsafe { core::str::from_utf8_unchecked(&self[read_offset..]) };
                let is_word_final =
                    final_sigma_automata.is_accepting() && !case_ignorable_then_cased(rest.chars());
                let sigma_lowercase = if is_word_final { 'ς' } else { 'σ' };
                encode_to_slice_or_else_to_queue(
                    sigma_lowercase,
                    &mut queue,
                    &mut self[..read_offset],
                    &mut write_offset,
                );
            } else {
                for c in uppercase_char.to_lowercase() {
                    encode_to_slice_or_else_to_queue(
                        c,
                        &mut queue,
                        &mut self[..read_offset],
                        &mut write_offset,
                    );
                }
            }
            final_sigma_automata.step(uppercase_char);
        }
        assert_eq!(read_offset, self.len());
        return if write_offset < read_offset { Ok(write_offset) } else { Err(queue) };

        // For now this is copy pasted from core::str, FIXME: DRY
        fn case_ignorable_then_cased<I: Iterator<Item = char>>(iter: I) -> bool {
            use core::unicode::{Case_Ignorable, Cased};
            match iter.skip_while(|&c| Case_Ignorable(c)).next() {
                Some(c) => Cased(c),
                None => false,
            }
        }
    }
}

fn encode_to_slice_or_else_to_queue(
    c: char,
    queue: &mut VecDeque<u8>,
    slice: &mut [u8],
    write_offset: &mut usize,
) {
    let mut buffer = [0; 4];
    let len = c.encode_utf8(&mut buffer).len();
    let writable_slice = &mut slice[*write_offset..];
    let direct_copy_length = core::cmp::min(len, writable_slice.len());
    writable_slice[..direct_copy_length].copy_from_slice(&buffer[..direct_copy_length]);
    *write_offset += direct_copy_length;
    queue.extend(&buffer[direct_copy_length..len]);
}

fn dump_queue(queue: &mut VecDeque<u8>, slice: &mut [u8], write_offset: &mut usize) {
    while *write_offset < slice.len() {
        match queue.pop_front() {
            Some(b) => {
                slice[*write_offset] = b;
                *write_offset += 1;
            }
            None => break,
        }
    }
}

#[derive(Clone)]
enum FinalSigmaAutomata {
    Init,
    Accepted,
}

impl FinalSigmaAutomata {
    fn new() -> Self {
        Self::Init
    }

    fn is_accepting(&self) -> bool {
        match self {
            FinalSigmaAutomata::Accepted => true,
            FinalSigmaAutomata::Init => false,
        }
    }

    fn step(&mut self, c: char) {
        use core::unicode::{Case_Ignorable, Cased};

        use FinalSigmaAutomata::*;
        *self = match self {
            Init => {
                if Cased(c) {
                    Accepted
                } else {
                    Init
                }
            }
            Accepted => {
                if Cased(c) || Case_Ignorable(c) {
                    Accepted
                } else {
                    Init
                }
            }
        }
    }
}
