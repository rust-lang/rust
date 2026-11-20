struct Texture;

trait Surface {
    type TextureIter<'a>: Iterator<Item = &'a Texture>
    where
        Self: 'a;

    fn get_texture(&self) -> Self::TextureIter<'_>;
}

trait Swapchain {
    type Surface<'a>: Surface
    where
        Self: 'a;

    fn get_surface(&self) -> Self::Surface<'_>;
}

impl<'s> Surface for &'s Texture {
    type TextureIter<'a> = std::option::IntoIter<&'a Texture>;
    //~^ ERROR the type

    fn get_texture(&self) -> Self::TextureIter<'_> {
        let option: Option<&Texture> = Some(self);
        option.into_iter()
    }
}

impl Swapchain for Texture {
    type Surface<'a> = &'a Texture;

    fn get_surface(&self) -> Self::Surface<'_> {
        self
    }
}

fn main() {}
