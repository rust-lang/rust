//@ revisions: ascii unicode
//@ compile-flags: --diagnostic-width=145
//@[unicode] compile-flags: -Zunstable-options --error-format=human-unicode
// ignore-tidy-linelength

fn main() {
    let unicode_is_fun = "؁‱ஹ௸௵꧄.ဪ꧅⸻𒈙𒐫﷽𒌄𒈟𒍼𒁎𒀱𒌧𒅃 𒈓𒍙𒊎𒄡𒅌𒁏𒀰𒐪𒐩𒈙𒐫𪚥";
    let _ = "👨👩👧👦👨👩👧👦👨👩👧👦👨👩👧👦👨👩👧👦👨👩👧👦👨👩👧👦👨👩👧👦👨👩👧👦👨👩👧👦👨👩👧👦👨👩👧👦👨👩👧👦👨👩👧👦👨👩👧👦👨👩👧👦👨👩👧👦👨👩👧👦👨👩👧👦👨👩👧👦👨👩👧👦👨👩👧👦👨👩👧👦👨👩👧👦👨👩👧👦👨👩👧👦👨👩👧👦👨👩👧👦👨👩👧👦👨👩👧👦👨👩👧👦👨👩👧👦👨👩👧👦👨👩👧👦👨👩👧👦👨👩👧👦👨👩👧👦👨👩👧👦👨👩👧👦👨👩👧👦👨👩👧👦👨👩👧👦👨👩👧👦👨👩👧👦👨👩👧👦👨👩👧👦👨👩👧👦👨👩👧👦👨👩👧👦"; let _a = unicode_is_fun + " really fun!";
    //[ascii]~^ ERROR cannot add `&str` to `&str`
    let _ = "👨‍👩‍👧‍👦👨‍👩‍👧‍👦👨‍👩‍👧‍👦👨‍👩‍👧‍👦👨‍👩‍👧‍👦👨‍👩‍👧‍👦👨‍👩‍👧‍👦👨‍👩‍👧‍👦👨‍👩‍👧‍👦👨‍👩‍👧‍👦👨‍👩‍👧‍👦👨‍👩‍👧‍👦👨‍👩‍👧‍👦👨‍👩‍👧‍👦👨‍👩‍👧‍👦👨‍👩‍👧‍👦👨‍👩‍👧‍👦👨‍👩‍👧‍👦👨‍👩‍👧‍👦👨‍👩‍👧‍👦👨‍👩‍👧‍👦👨‍👩‍👧‍👦👨‍👩‍👧‍👦👨‍👩‍👧‍👦👨‍👩‍👧‍👦👨‍👩‍👧‍👦👨‍👩‍👧‍👦👨‍👩‍👧‍👦👨‍👩‍👧‍👦👨‍👩‍👧‍👦👨‍👩‍👧‍👦👨‍👩‍👧‍👦👨‍👩‍👧‍👦👨‍👩‍👧‍👦👨‍👩‍👧‍👦👨‍👩‍👧‍👦👨‍👩‍👧‍👦👨‍👩‍👧‍👦👨‍👩‍👧‍👦👨‍👩‍👧‍👦👨‍👩‍👧‍👦👨‍👩‍👧‍👦👨‍👩‍👧‍👦👨‍👩‍👧‍👦👨‍👩‍👧‍👦👨‍👩‍👧‍👦👨‍👩‍👧‍👦👨‍👩‍👧‍👦👨‍👩‍👧‍👦"; let _a = unicode_is_fun + " really fun!";
    //[ascii]~^ ERROR cannot add `&str` to `&str`
    let _ = "ༀ༁༂༃༄༅༆༇༈༉༊་༌།༎༏༐༑༒༓༔༕༖༗༘༙༚༛༜༝༞༟༠༡༢༣༤༥༦༧༨༩༪༫༬༭༮༯༰༱༲༳༴༵༶༷༸༹༺༻༼༽༾༿ཀཁགགྷངཅཆཇ཈ཉཊཋཌཌྷཎཏཐདདྷནཔཕབབྷམཙཚཛཛྷཝཞཟའཡརལཤཥསཧཨཀྵཪཫཬ཭཮཯཰ཱཱཱིིུུྲྀཷླྀཹེཻོཽཾཿ྄ཱྀྀྂྃ྅྆྇ྈྉྊྋྌྍྎྏྐྑྒྒྷྔྕྖྗ྘ྙྚྛྜྜྷྞྟྠྡྡྷྣྤྥྦྦྷྨྩྪྫྫྷྭྮྯྰྱྲླྴྵྶྷྸྐྵྺྻྼ྽྾྿࿀࿁࿂࿃࿄࿅࿆࿇࿈࿉࿊࿋࿌࿍࿎࿏࿐࿑࿒࿓࿔࿕࿖࿗࿘࿙࿚"; let _a = unicode_is_fun + " really fun!";
    //[ascii]~^ ERROR cannot add `&str` to `&str`
    let _ = "xxxxxxx👨‍👩‍👧‍👦xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx👨xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"; let _a = unicode_is_fun + " really fun!";
    //[ascii]~^ ERROR cannot add `&str` to `&str`
}
