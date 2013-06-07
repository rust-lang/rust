<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="1.0">
  <xsl:param name="funcsynopsis.style">ansi</xsl:param>
  <xsl:param name="function.parens" select="1"/>
  <xsl:template match="mallctl">
    "<xsl:call-template name="inline.monoseq"/>"
  </xsl:template>
</xsl:stylesheet>
