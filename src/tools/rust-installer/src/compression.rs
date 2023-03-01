use anyhow::{Context, Error};
use flate2::{read::GzDecoder, write::GzEncoder};
use rayon::prelude::*;
use std::{convert::TryFrom, fmt, io::Read, io::Write, path::Path, str::FromStr};
use xz2::{read::XzDecoder, write::XzEncoder};

#[derive(Debug, Copy, Clone)]
pub enum CompressionFormat {
    Gz,
    Xz,
}

impl CompressionFormat {
    pub(crate) fn detect_from_path(path: impl AsRef<Path>) -> Option<Self> {
        match path.as_ref().extension().and_then(|e| e.to_str()) {
            Some("gz") => Some(CompressionFormat::Gz),
            Some("xz") => Some(CompressionFormat::Xz),
            _ => None,
        }
    }

    pub(crate) fn extension(&self) -> &'static str {
        match self {
            CompressionFormat::Gz => "gz",
            CompressionFormat::Xz => "xz",
        }
    }

    pub(crate) fn encode(&self, path: impl AsRef<Path>) -> Result<Box<dyn Encoder>, Error> {
        let mut os = path.as_ref().as_os_str().to_os_string();
        os.push(format!(".{}", self.extension()));
        let path = Path::new(&os);

        if path.exists() {
            crate::util::remove_file(path)?;
        }
        let file = crate::util::create_new_file(path)?;

        Ok(match self {
            CompressionFormat::Gz => Box::new(GzEncoder::new(file, flate2::Compression::best())),
            CompressionFormat::Xz => {
                let mut filters = xz2::stream::Filters::new();
                // the preset is overridden by the other options so it doesn't matter
                let mut lzma_ops = xz2::stream::LzmaOptions::new_preset(9).unwrap();
                // This sets the overall dictionary size, which is also how much memory (baseline)
                // is needed for decompression.
                lzma_ops.dict_size(64 * 1024 * 1024);
                // Use the best match finder for compression ratio.
                lzma_ops.match_finder(xz2::stream::MatchFinder::BinaryTree4);
                lzma_ops.mode(xz2::stream::Mode::Normal);
                // Set nice len to the maximum for best compression ratio
                lzma_ops.nice_len(273);
                // Set depth to a reasonable value, 0 means auto, 1000 is somwhat high but gives
                // good results.
                lzma_ops.depth(1000);
                // 2 is the default and does well for most files
                lzma_ops.position_bits(2);
                // 0 is the default and does well for most files
                lzma_ops.literal_position_bits(0);
                // 3 is the default and does well for most files
                lzma_ops.literal_context_bits(3);

                filters.lzma2(&lzma_ops);

                let mut builder = xz2::stream::MtStreamBuilder::new();
                builder.filters(filters);

                // On 32-bit platforms limit ourselves to 3 threads, otherwise we exceed memory
                // usage this process can take. In the future we'll likely only do super-fast
                // compression in CI and move this heavyweight processing to promote-release (which
                // is always 64-bit and can run on big-memory machines) but for now this lets us
                // move forward.
                if std::mem::size_of::<usize>() == 4 {
                    builder.threads(3);
                } else {
                    builder.threads(6);
                }

                let compressor = XzEncoder::new_stream(
                    std::io::BufWriter::new(file),
                    builder.encoder().unwrap(),
                );
                Box::new(compressor)
            }
        })
    }

    pub(crate) fn decode(&self, path: impl AsRef<Path>) -> Result<Box<dyn Read>, Error> {
        let file = crate::util::open_file(path.as_ref())?;
        Ok(match self {
            CompressionFormat::Gz => Box::new(GzDecoder::new(file)),
            CompressionFormat::Xz => Box::new(XzDecoder::new(file)),
        })
    }
}

/// This struct wraps Vec<CompressionFormat> in order to parse the value from the command line.
#[derive(Debug, Clone)]
pub struct CompressionFormats(Vec<CompressionFormat>);

impl TryFrom<&'_ str> for CompressionFormats {
    type Error = Error;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        let mut parsed = Vec::new();
        for format in value.split(',') {
            match format.trim() {
                "gz" => parsed.push(CompressionFormat::Gz),
                "xz" => parsed.push(CompressionFormat::Xz),
                other => anyhow::bail!("unknown compression format: {}", other),
            }
        }
        Ok(CompressionFormats(parsed))
    }
}

impl FromStr for CompressionFormats {
    type Err = Error;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::try_from(value)
    }
}

impl fmt::Display for CompressionFormats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, format) in self.iter().enumerate() {
            if i != 0 {
                write!(f, ",")?;
            }
            fmt::Display::fmt(
                match format {
                    CompressionFormat::Xz => "xz",
                    CompressionFormat::Gz => "gz",
                },
                f,
            )?;
        }
        Ok(())
    }
}

impl Default for CompressionFormats {
    fn default() -> Self {
        Self(vec![CompressionFormat::Gz, CompressionFormat::Xz])
    }
}

impl CompressionFormats {
    pub(crate) fn iter(&self) -> impl Iterator<Item = CompressionFormat> + '_ {
        self.0.iter().map(|i| *i)
    }
}

pub(crate) trait Encoder: Send + Write {
    fn finish(self: Box<Self>) -> Result<(), Error>;
}

impl<W: Send + Write> Encoder for GzEncoder<W> {
    fn finish(self: Box<Self>) -> Result<(), Error> {
        GzEncoder::finish(*self).context("failed to finish .gz file")?;
        Ok(())
    }
}

impl<W: Send + Write> Encoder for XzEncoder<W> {
    fn finish(self: Box<Self>) -> Result<(), Error> {
        XzEncoder::finish(*self).context("failed to finish .xz file")?;
        Ok(())
    }
}

pub(crate) struct CombinedEncoder {
    encoders: Vec<Box<dyn Encoder>>,
}

impl CombinedEncoder {
    pub(crate) fn new(encoders: Vec<Box<dyn Encoder>>) -> Box<dyn Encoder> {
        Box::new(Self { encoders })
    }
}

impl Write for CombinedEncoder {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.write_all(buf)?;
        Ok(buf.len())
    }

    fn write_all(&mut self, buf: &[u8]) -> std::io::Result<()> {
        self.encoders
            .par_iter_mut()
            .map(|w| w.write_all(buf))
            .collect::<std::io::Result<Vec<()>>>()?;
        Ok(())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.encoders
            .par_iter_mut()
            .map(|w| w.flush())
            .collect::<std::io::Result<Vec<()>>>()?;
        Ok(())
    }
}

impl Encoder for CombinedEncoder {
    fn finish(self: Box<Self>) -> Result<(), Error> {
        self.encoders
            .into_par_iter()
            .map(|e| e.finish())
            .collect::<Result<Vec<()>, Error>>()?;
        Ok(())
    }
}
